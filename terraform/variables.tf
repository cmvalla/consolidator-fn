variable "project_id" {
  type        = string
  description = "The GCP project ID."
}

variable "region" {
  type        = string
  description = "The GCP region."
}

variable "location" {
  type        = string
  description = "The GCP location."
}

variable "repository_id" {
  type        = string
  description = "The Artifact Registry repository ID."
}

variable "image_name" {
  type        = string
  description = "The name of the Docker image."
}

variable "image_tag" {
  type        = string
  description = "The tag of the Docker image, typically the Build ID."
}

variable "topic_name" {
  type        = string
  description = "The Pub/Sub topic name that triggers the consolidator."
}

variable "consolidator_sa_email" {
  type        = string
  description = "The email of the consolidator service account."
}

variable "consolidator_sa_roles" {
  type = list(string)
  description = "Project-level IAM roles for the consolidator service account"
  default = [
    "roles/run.invoker",
    "roles/eventarc.eventReceiver",
    "roles/serviceusage.serviceUsageConsumer",
    "roles/cloudtrace.agent"
  ]
}
