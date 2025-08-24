terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0.0"
    }
  }
}

provider "google" {
  credentials = "${file("../../../credentials.json")}"
  project = var.project_id
  region  = var.region
}
