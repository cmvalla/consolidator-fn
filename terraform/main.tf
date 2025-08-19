resource "google_cloud_run_v2_service" "consolidator" {
  project  = var.project_id
  name     = "consolidator-fn"
  location = var.location

  template {
    service_account = var.consolidator_sa_email
    timeout         = "1800s" # 30 minutes for potentially long-running consolidations
    scaling {
      min_instance_count = 0 # Can scale to zero
      max_instance_count = 3
    }
    containers {
      image = "${var.location}-docker.pkg.dev/${var.project_id}/${var.repository_id}/${var.image_name}:${var.image_tag}"
      ports {
        container_port = 8080
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      # Add other necessary environment variables here, e.g., for Spanner instance and database names
    }
  }
}

resource "google_eventarc_trigger" "consolidator_trigger" {
  project  = var.project_id
  name     = "consolidator-trigger-pubsub"
  location = var.location

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.pubsub.topic.v1.messagePublished"
  }

  destination {
    cloud_run_service {
      service = google_cloud_run_v2_service.consolidator.name
      region  = var.location
    }
  }

  transport {
    pubsub {
      topic = "projects/${var.project_id}/topics/${var.topic_name}"
    }
  }

  service_account = var.consolidator_sa_email
}
