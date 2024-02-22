from datasets import load_dataset, load_metric
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import torch
import numpy as np

class Training:
    def __init__(self, dataPath, outdir, modelNameOrPath = 'google/vit-base-patch16-224-in21k', metricName="accuracy") :
        self.featureExtractor = ViTImageProcessor.from_pretrained(modelNameOrPath)
        self.dataset = load_dataset("imagefolder", data_dir=dataPath)
        self.preparedDs = self.dataset.with_transform(self.transform)
        print(self.preparedDs['train'][0:2]["pixel_values"].shape)
        self.labels = self.dataset['train'].features['label'].names
        self.model = self.initModel(modelNameOrPath)
        self.mectric = load_metric(metricName)
        self.trainer = self.initTrainer(outdir)

    def transform(self, example_batch):
        inputs = self.featureExtractor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')

        inputs['labels'] = example_batch['label']
        return inputs

    def collate_fn(self, batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
 
    def compute_metrics(self, p):
        return self.metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    def initModel(self, modelNameOrPath) :
        model = ViTForImageClassification.from_pretrained(
            modelNameOrPath,
            num_labels=len(self.labels),
            id2label={str(i): c for i, c in enumerate(self.labels)},
            label2id={c: str(i) for i, c in enumerate(self.labels)}
            )
        return model

    def initTrainer(self, outdir) :
        training_args = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=10,
        fp16=False,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=self.preparedDs["train"],
            eval_dataset=self.preparedDs["test"],
            tokenizer=self.featureExtractor,
        )
        
        return trainer
    
    def trainModel(self) :
        train_results = self.trainer.train()
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_results.metrics)
        self.trainer.save_metrics("train", train_results.metrics)
        self.trainer.save_state()

        metrics = self.trainer.evaluate(self.preparedDs['validation'])
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
