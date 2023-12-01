import math

def do_evaluation(trainer, data_args, eval_dataset):

    metrics = trainer.evaluate()

    max_eval_samples = (
        data_args.max_eval_samples
        if data_args.max_eval_samples is not None
        else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return trainer
