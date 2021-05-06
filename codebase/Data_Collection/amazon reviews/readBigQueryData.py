from google.cloud import bigquery



def query_bq():
    client = bigquery.Client()
    query_job = client.query(
       """
       select * from `ecstatic-armor-297504.amazon_reviews.cell_reviews` 
       limit 10
       """
       # Query to be written 
    )

    results = query_job.result()  # Waits for job to complete.

    for row in results:
        print("{} : {} views".format(row.overall, row.reviewText))


query_bq()