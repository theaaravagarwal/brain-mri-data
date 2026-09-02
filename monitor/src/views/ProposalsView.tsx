import "./views.css";

export type FrozenJob = { runId: string; profile: string; state: string; reasonCode: string; proposalAllowed: boolean };
export type JobProposal = { runId: string | null; profile: string | null; state: "exact-match" | "abstained" | "unavailable" | "unsafe" | "already-complete"; reason: string; executed: false };

export type ProposalsViewModel = { jobs: FrozenJob[]; proposals: JobProposal[] };

export function ProposalsView({ data }: { data: ProposalsViewModel | null }) {
  if (!data) return <section className="view-shell" aria-labelledby="proposals-title"><div className="view-heading"><div><h1 id="proposals-title">Proposals</h1></div></div><div className="view-state view-state--empty"><h2>No proposal data</h2><p>Validated job status has not been received.</p></div></section>;
  return <section className="view-shell" aria-labelledby="proposals-title">
    <div className="view-heading"><div><h1 id="proposals-title">Proposals</h1><p className="view-lede">Read-only availability from frozen job contracts. No proposal can start or mutate a worker.</p></div><span className="view-status view-status--pending">Read only</span></div>
    <p className="view-disclaimer" role="note">Proposal awaiting human review. Every displayed proposal is non-executing.</p>
    <article className="view-panel"><div className="view-panel__heading"><h2>Frozen jobs</h2><span className="view-helper">Allowlisted run/profile pairs</span></div><div className="view-table-wrap"><table className="view-table"><caption className="sr-only">Frozen jobs and proposal eligibility</caption><thead><tr><th scope="col">Run ID</th><th scope="col">Profile</th><th scope="col">State</th><th scope="col">Reason code</th><th scope="col">Proposal</th></tr></thead><tbody>{data.jobs.length ? data.jobs.map(job => <tr key={`${job.runId}-${job.profile}`}><th scope="row"><code>{job.runId}</code></th><td><code>{job.profile}</code></td><td>{job.state}</td><td>{job.reasonCode}</td><td><span className={`view-status view-status--${job.proposalAllowed ? "valid" : "blocked"}`}>{job.proposalAllowed ? "Allowed" : "Blocked"}</span></td></tr>) : <tr><td colSpan={5} className="view-empty-cell">No frozen jobs available.</td></tr>}</tbody></table></div></article>
    <article className="view-panel"><div className="view-panel__heading"><h2>Review proposals</h2><span className="view-helper">No execution controls</span></div><div className="view-proposal-list">{data.proposals.length ? data.proposals.map((proposal, index) => <div className="view-proposal" key={`${proposal.runId ?? "none"}-${proposal.profile ?? "none"}-${index}`}><div><h3>{proposal.runId && proposal.profile ? <><code>{proposal.runId}</code><span aria-hidden="true"> · </span><code>{proposal.profile}</code></> : `Proposal ${index + 1}: no exact match`}</h3><p>{proposal.reason}</p></div><div className="view-proposal__meta"><strong className="view-executed">executed: false</strong><span className={`view-status view-status--${proposal.state}`}>{proposal.state}</span></div></div>) : <div className="view-state view-state--empty"><h3>No proposal available</h3><p>There is no exact, pre-approved run/profile match to display.</p></div>}</div></article>
  </section>;
}

export default ProposalsView;
