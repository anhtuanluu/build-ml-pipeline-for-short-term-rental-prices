#!/usr/bin/env python
"""
a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info("Creating basic cleaning run")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info('Read artifact')
    df = pd.read_csv(artifact_local_path)
    
    logger.info('Drop outlier')
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # add new fix
    # idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    # df = df[idx].copy()
    
    logger.info('Save csv')
    df.to_csv("clean_sample.csv", index=False)

    #Upload file csv to WanDB
    logger.info('Upload file csv to WanDB')

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type= str,
        help= "Name input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type= str,
        help= "Name output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type= str,
        help= "Output type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type= str,
        help= "A brief description of output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type= float,
        help= "The min of price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type= float,
        help= "The max of price",
        required=True
    )


    args = parser.parse_args()

    go(args)
