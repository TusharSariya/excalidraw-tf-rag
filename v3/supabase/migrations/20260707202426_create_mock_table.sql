create table mock_items (
  id bigint generated always as identity primary key,
  name text not null,
  created_at timestamptz not null default now()
);
