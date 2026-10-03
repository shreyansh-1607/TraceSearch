Operating Systems: Deadlocks and Scheduling

A deadlock is a situation in which processes wait for resources held by other waiting processes. The four necessary conditions are mutual exclusion, hold and wait, no preemption, and circular wait. Deadlock prevention breaks at least one of these conditions. Requiring all processes to acquire resources in a fixed global order prevents circular wait.

Starvation differs from deadlock. A runnable process can wait indefinitely when other jobs repeatedly receive priority. Aging gradually increases the priority of a waiting process, helping prevent starvation. It does not eliminate the resource cycles responsible for deadlock.

Round-robin scheduling gives each ready process a bounded time quantum. When the quantum expires, an unfinished process returns to the end of the ready queue. A shorter quantum can improve interactive response but increases context-switch overhead. First-come-first-served scheduling can make short jobs wait behind one long job.

Virtual memory divides an address space into pages. A page fault occurs when a referenced page is not currently resident in physical memory. The operating system resolves a valid fault by making the page available and resuming execution; an invalid access can terminate the process.

---

Networking: Streams, Addresses, and Recovery

TCP provides an ordered byte stream between connected endpoints. One send call does not correspond to one receive call. An application can prefix each message with its byte length, then loop until it has received that many bytes. This technique is called length-prefix framing.

An IP address identifies a network interface for routing, while a port distinguishes a service at an endpoint. The loopback IPv4 address 127.0.0.1 refers to the local computer. A server listening only on loopback cannot accept connections from another laptop over Wi-Fi.

TCP retransmission repairs losses during a live connection. It does not remember application progress after a program restarts. A resumable file-transfer application records an acknowledged byte offset and verifies the saved prefix before sending the remaining data.

TLS encrypts network traffic and authenticates endpoints according to the configured trust rules. A content hash such as SHA-256 detects differences relative to an expected digest; hashing alone does not establish who supplied the digest. Certificate verification and hashing serve different purposes.

---

# Database Transactions and Indexes

A transaction groups database operations into a unit that either commits or rolls back. Atomicity means incomplete operations do not leave a partially committed transaction. Durability means committed changes are retained according to the database's configured durability guarantees.

For seat reservations, a unique constraint on the event and seat identifiers prevents two committed bookings for the same seat. Checking availability and later inserting without a constraint or suitable transaction handling creates a race condition. The application must handle a rejected conflicting booking.

An index can reduce the records examined for a selective query, at the cost of extra storage and maintenance during writes. An index is not automatically faster for every query. EXPLAIN describes the chosen execution plan; repeated measurements should account for caching and dataset size.

Idempotency means retrying an operation has the same intended effect as performing it once. A unique request identifier can help an order service recognize a repeated request rather than deducting inventory a second time. The identifier and corresponding effect must be recorded consistently.

---

Semantic Search and Incremental Indexing

An embedding represents a passage as a numerical vector. Semantic retrieval compares a query vector with passage vectors and ranks nearby passages. Similarity scores are not calibrated probabilities that a passage answers a question. Nearest-neighbor search can return results even when the collection contains no answer.

An incremental indexer compares stored content hashes with the current files. New files are processed, changed files replace their old searchable chunks, deleted files are excluded, and unchanged files reuse existing embeddings. Rebuilding a FAISS index from saved vectors is different from recomputing those vectors with a model.

A source reference records a document path and a page or line location alongside each chunk. The numerical vector alone is insufficient to display evidence. A metadata database maps retrieved vector identifiers back to text and source locations.

To publish an update safely, prepare a complete new index generation before changing the active database pointer. If preparation fails, the previous generation remains usable. A model revision or chunking configuration change requires compatible reindexing rather than mixing vectors from different embedding spaces.
