variable "project_id" {
  description = "The Google Cloud project ID."
  type        = string
}

variable "location" {
  description = "The GCP region for the Cloud Run service."
  type        = string
}

variable "region" {
  description = "The GCP region for the Cloud Run service."
  type        = string
}

variable "consolidator_sa_email" {
  description = "Service account email for the consolidator function."
  type        = string
}

variable "vpc_connector" {
  description = "The VPC Access connector to use for the Cloud Run service."
  type        = string
}

variable "image_url" {
  description = "The URL of the Docker image for the consolidator function."
  type        = string
}

variable "image_tag" {
  description = "The tag for the Docker image."
  type        = string
}

variable "redis_host" {
  description = "The Redis host."
  type        = string
}

variable "redis_port" {
  description = "The Redis port."
  type        = number
}

variable "spanner_instance_id" {
  description = "The Spanner instance ID."
  type        = string
}

variable "spanner_database_id" {
  description = "The Spanner database ID."
  type        = string
}

variable "gemini_api_key_secret_id" {
  description = "The Secret Manager ID for the Gemini API key."
  type        = string
}

variable "embedding_service_url" {
  description = "The URL of the embedding service."
  type        = string
}

variable "persistor_topic_name" {
  description = "The name of the Pub/Sub topic for the persistor function."
  type        = string
}

variable "consolidator_max_workers" {
  description = "The maximum number of worker threads for the consolidator function."
  type        = number
}

variable "llm_batch_size" {
  description = "The batch size for LLM calls in the consolidator function."
  type        = number
}

variable "topic_name" {
  description = "The name of the Pub/Sub topic for the consolidator function."
  type        = string
}

variable "use_gemini_embeddings" {
  description = "Set to true to use Gemini Embeddings (gemini-embeddings-001) instead of the external embedding service."
  type        = bool
}

variable "llm_model_name" {
  description = "The name of the LLM model to use for the consolidator function."
  type        = string
}