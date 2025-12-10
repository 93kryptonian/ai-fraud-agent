drop table if exists documents cascade;

create table documents (
    id uuid primary key,
    source_name text not null,
    page int not null,
    content text not null,
    hash text unique not null,
    created_at timestamptz default now()
);
drop table if exists document_embeddings cascade;

create table document_embeddings (
    id uuid primary key,
    document_id uuid not null references documents(id) on delete cascade,
    embedding vector(768) not null,
    created_at timestamptz default now()
);

create index on document_embeddings using hnsw (embedding vector_cosine_ops);
create index documents_source_idx on documents(source_name);
create index documents_page_idx on documents(page);


create or replace function match_documents(
  filter jsonb,
  query_embedding vector(768)
)
returns table (
  id uuid,
  content text,
  page int,
  source_name text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    d.id,
    doc.content,
    doc.page,
    doc.source_name,
    1 - (d.embedding <=> query_embedding) as similarity
  from document_embeddings d
  join documents doc on doc.id = d.document_id
  where 
    (filter is null 
     or (filter ? 'source_name' and doc.source_name = filter->>'source_name'))
  order by d.embedding <=> query_embedding
  limit 50;  -- LangChain will trim top_k itself
end;
$$;
