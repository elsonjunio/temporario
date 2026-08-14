"use strict";

console.log("app iniciado");
console.info("info do app");
console.debug("debug do app");
console.warn("aviso do app");

fetch("/api/ping")
  .then((r) => r.json())
  .then((d) => console.log("ping:", d.status))
  .catch((e) => console.error("ping falhou", e));

async function loadUsers() {
  try {
    const res = await fetch("/api/users");
    const data = await res.json();
    const body = document.getElementById("users-body");
    body.innerHTML = "";
    data.users.forEach((u) => {
      const tr = document.createElement("tr");
      const td1 = document.createElement("td");
      td1.textContent = u.name;
      const td2 = document.createElement("td");
      td2.textContent = u.email;
      tr.append(td1, td2);
      body.append(tr);
    });
  } catch (e) {
    console.error("erro ao carregar usuários", e);
  }
}
loadUsers();

const form = document.getElementById("user-form");
form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const payload = {
    name: document.getElementById("name").value,
    email: document.getElementById("email").value,
    role: document.getElementById("role").value,
  };
  const result = document.getElementById("form-result");
  try {
    const res = await fetch("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    result.textContent = res.ok ? `Sucesso: ${data.message}` : `Erro: ${data.error}`;
    console.log("POST /api/users ->", res.status);
  } catch (e) {
    result.textContent = "Falha de rede";
    console.error("POST falhou", e);
  }
});

document.getElementById("toggle-panel").addEventListener("click", () => {
  const panel = document.getElementById("dynamic-panel");
  const hidden = panel.classList.toggle("hidden");
  console.log("painel " + (hidden ? "oculto" : "visível"));
});

document.getElementById("boom").addEventListener("click", () => {
  throw new TypeError("falha intencional do JS");
});
