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

variable "topic_resource_id" {
  type        = string
  description = "The resource ID of the Pub/Sub topic that triggers the consolidator."
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

variable "redis_host" {
  type        = string
  description = "The Redis host."
}

variable "redis_port" {
  type        = string
  description = "The Redis port."
}



variable "memgraph_host" {
  type        = string
  description = "The Memgraph host."
}

variable "memgraph_port" {
  type        = string
  description = "The Memgraph port."
}

variable "memgraph_password" {
  type        = string
  description = "The Memgraph password."
  sensitive   = true
}

variable "vpc_connector" {
  type        = string
  description = "The Serverless VPC Access connector name."
}
variable "topic_name" {
  type        = string
  description = "The name of the Pub/Sub topic."
}