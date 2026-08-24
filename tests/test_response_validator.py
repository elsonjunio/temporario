from __future__ import annotations

import unittest

from src.response_validator import classify_response_intent, should_reprompt


class TestClassifyResponseIntent(unittest.TestCase):
    # ------------------------------------------------------------------
    # Thinking patterns – PT-BR
    # ------------------------------------------------------------------
    def test_thinking_pt_vou_verificar(self):
        self.assertEqual(
            classify_response_intent("Vou verificar o conteúdo do arquivo."),
            "thinking",
        )

    def test_thinking_pt_deixe_me(self):
        self.assertEqual(
            classify_response_intent("Deixe-me analisar esse código."),
            "thinking",
        )

    def test_thinking_pt_deixa_eu(self):
        self.assertEqual(
            classify_response_intent("Deixa eu ver o que temos aqui."),
            "thinking",
        )

    def test_thinking_pt_agora_vou(self):
        self.assertEqual(
            classify_response_intent("Agora vou ler o arquivo de configuração."),
            "thinking",
        )

    def test_thinking_pt_preciso_ver(self):
        self.assertEqual(
            classify_response_intent("Preciso verificar se o arquivo existe."),
            "thinking",
        )

    def test_thinking_pt_vou_comecar(self):
        self.assertEqual(
            classify_response_intent("Vou começar listando os arquivos do projeto."),
            "thinking",
        )

    def test_thinking_pt_primeiro_vou(self):
        self.assertEqual(
            classify_response_intent("Primeiro, vou buscar os arquivos relevantes."),
            "thinking",
        )

    def test_thinking_pt_antes_de(self):
        self.assertEqual(
            classify_response_intent(
                "Antes de fazer qualquer coisa, vou analisar o código."
            ),
            "thinking",
        )

    def test_thinking_pt_vou_usar_tool(self):
        # "Vou usar read_file" matches both thinking and tool_hint patterns,
        # but tool_hint is more specific and checked first.
        self.assertEqual(
            classify_response_intent("Vou usar read_file para ver o conteúdo."),
            "tool_hint",
        )

    # ------------------------------------------------------------------
    # Thinking patterns – EN
    # ------------------------------------------------------------------
    def test_thinking_en_let_me_check(self):
        self.assertEqual(
            classify_response_intent("Let me check the file contents."),
            "thinking",
        )

    def test_thinking_en_ill_check(self):
        self.assertEqual(
            classify_response_intent("I'll check the configuration file."),
            "thinking",
        )

    def test_thinking_en_now_ill(self):
        self.assertEqual(
            classify_response_intent("Now I'll read the source code."),
            "thinking",
        )

    def test_thinking_en_i_need_to(self):
        self.assertEqual(
            classify_response_intent("I need to verify the file exists first."),
            "thinking",
        )

    def test_thinking_en_first_let_me(self):
        self.assertEqual(
            classify_response_intent("First, let me search for the relevant files."),
            "thinking",
        )

    def test_thinking_en_before_doing(self):
        self.assertEqual(
            classify_response_intent(
                "Before doing anything, let me inspect the codebase."
            ),
            "thinking",
        )

    def test_thinking_en_ill_first(self):
        self.assertEqual(
            classify_response_intent("I'll first list the directory contents."),
            "thinking",
        )

    # ------------------------------------------------------------------
    # Planning patterns – PT-BR
    # ------------------------------------------------------------------
    def test_planning_pt_meu_plano(self):
        self.assertEqual(
            classify_response_intent(
                "Meu plano é analisar os arquivos e depois fazer as alterações."
            ),
            "planning",
        )

    def test_planning_pt_abordagem(self):
        self.assertEqual(
            classify_response_intent(
                "A abordagem seria usar grep_files para localizar os erros."
            ),
            "planning",
        )

    def test_planning_pt_para_resolver(self):
        self.assertEqual(
            classify_response_intent(
                "Para resolver isso, precisamos criar um novo arquivo de configuração."
            ),
            "planning",
        )

    def test_planning_pt_melhor_seria(self):
        self.assertEqual(
            classify_response_intent(
                "O melhor seria refatorar o módulo principal primeiro."
            ),
            "planning",
        )

    # ------------------------------------------------------------------
    # Planning patterns – EN
    # ------------------------------------------------------------------
    def test_planning_en_my_plan(self):
        self.assertEqual(
            classify_response_intent(
                "My plan is to analyze the files and then make changes."
            ),
            "planning",
        )

    def test_planning_en_approach(self):
        self.assertEqual(
            classify_response_intent(
                "The approach would be to use grep_files to find the errors."
            ),
            "planning",
        )

    def test_planning_en_to_solve(self):
        self.assertEqual(
            classify_response_intent(
                "To solve this, we need to create a new config file."
            ),
            "planning",
        )

    def test_planning_en_herell_what_ill_do(self):
        self.assertEqual(
            classify_response_intent(
                "Here's what I'll do: first list the directory, then read each file."
            ),
            "planning",
        )

    # ------------------------------------------------------------------
    # Tool hint patterns – PT-BR
    # ------------------------------------------------------------------
    def test_tool_hint_pt_vou_usar(self):
        self.assertEqual(
            classify_response_intent("Vou usar read_file para verificar o conteúdo."),
            "tool_hint",
        )

    def test_tool_hint_pt_deveria_chamar(self):
        self.assertEqual(
            classify_response_intent("Deveria chamar grep_files para buscar os erros."),
            "tool_hint",
        )

    def test_tool_hint_pt_devemos_executar(self):
        self.assertEqual(
            classify_response_intent(
                "Devemos executar run_command para compilar o projeto."
            ),
            "tool_hint",
        )

    # ------------------------------------------------------------------
    # Tool hint patterns – EN
    # ------------------------------------------------------------------
    def test_tool_hint_en_ill_use(self):
        self.assertEqual(
            classify_response_intent("I'll use read_file to check the contents."),
            "tool_hint",
        )

    def test_tool_hint_en_should_call(self):
        self.assertEqual(
            classify_response_intent("We should call grep_files to search for errors."),
            "tool_hint",
        )

    # ------------------------------------------------------------------
    # Stalling patterns
    # ------------------------------------------------------------------
    def test_stalling_hmm(self):
        self.assertEqual(classify_response_intent("Hmm..."), "stalling")

    def test_stalling_well(self):
        self.assertEqual(classify_response_intent("Well..."), "stalling")

    def test_stalling_bem(self):
        self.assertEqual(classify_response_intent("Bem..."), "stalling")

    def test_stalling_interessante(self):
        self.assertEqual(classify_response_intent("Interessante."), "stalling")

    def test_stalling_deixe_me_pensar(self):
        self.assertEqual(classify_response_intent("Deixe-me pensar..."), "stalling")

    def test_stalling_let_me_think(self):
        self.assertEqual(classify_response_intent("Let me think..."), "stalling")

    def test_stalling_very_short(self):
        # Short responses that are NOT in the stalling patterns are treated as
        # answers (e.g. "Ok" can be a legitimate acknowledgment).
        self.assertEqual(classify_response_intent("Hmm"), "stalling")

    def test_stalling_empty(self):
        self.assertEqual(classify_response_intent(""), "stalling")

    # ------------------------------------------------------------------
    # Phantom action patterns – PT-BR
    # ------------------------------------------------------------------
    def test_phantom_pt_criei(self):
        self.assertEqual(
            classify_response_intent("Criei o arquivo main.py com o conteúdo correto."),
            "phantom_action",
        )

    def test_phantom_pt_atualizei(self):
        self.assertEqual(
            classify_response_intent("Atualizei a configuração do projeto."),
            "phantom_action",
        )

    def test_phantom_pt_removi(self):
        self.assertEqual(
            classify_response_intent("Removi o bug do código de autenticação."),
            "phantom_action",
        )

    def test_phantom_pt_instalei(self):
        self.assertEqual(
            classify_response_intent("Instalei todas as dependências necessárias."),
            "phantom_action",
        )

    def test_phantom_pt_configurei(self):
        self.assertEqual(
            classify_response_intent("Configurei o ambiente de desenvolvimento."),
            "phantom_action",
        )

    def test_phantom_pt_pronto(self):
        self.assertEqual(
            classify_response_intent("Pronto, já criei o arquivo de configuração."),
            "phantom_action",
        )

    def test_phantom_pt_ja_criei(self):
        self.assertEqual(
            classify_response_intent("Já atualizei o script de deploy."),
            "phantom_action",
        )

    def test_phantom_pt_passive(self):
        self.assertEqual(
            classify_response_intent("O arquivo foi criado com sucesso."),
            "phantom_action",
        )

    def test_phantom_pt_passive_config(self):
        self.assertEqual(
            classify_response_intent("A configuração foi atualizada corretamente."),
            "phantom_action",
        )

    def test_phantom_pt_feito(self):
        self.assertEqual(
            classify_response_intent("Feito! O arquivo foi modificado."),
            "phantom_action",
        )

    # ------------------------------------------------------------------
    # Phantom action patterns – EN
    # ------------------------------------------------------------------
    def test_phantom_en_created(self):
        self.assertEqual(
            classify_response_intent(
                "I created the file main.py with the correct content."
            ),
            "phantom_action",
        )

    def test_phantom_en_updated(self):
        self.assertEqual(
            classify_response_intent("I updated the project configuration."),
            "phantom_action",
        )

    def test_phantom_en_deleted(self):
        self.assertEqual(
            classify_response_intent("I deleted the old authentication code."),
            "phantom_action",
        )

    def test_phantom_en_installed(self):
        self.assertEqual(
            classify_response_intent("I installed all the necessary dependencies."),
            "phantom_action",
        )

    def test_phantom_en_configured(self):
        self.assertEqual(
            classify_response_intent("I configured the development environment."),
            "phantom_action",
        )

    def test_phantom_en_done_created(self):
        self.assertEqual(
            classify_response_intent("Done, I've created the configuration file."),
            "phantom_action",
        )

    def test_phantom_en_has_created(self):
        self.assertEqual(
            classify_response_intent("The file has been created successfully."),
            "phantom_action",
        )

    def test_phantom_en_fixed(self):
        self.assertEqual(
            classify_response_intent("I fixed the bug in the authentication module."),
            "phantom_action",
        )

    def test_phantom_en_deployed(self):
        self.assertEqual(
            classify_response_intent("I deployed the application to production."),
            "phantom_action",
        )

    # ------------------------------------------------------------------
    # Phantom action – negation (should NOT be phantom)
    # ------------------------------------------------------------------
    def test_phantom_negation_pt(self):
        self.assertEqual(
            classify_response_intent("Não criei nenhum arquivo."),
            "answer",
        )

    def test_phantom_negation_pt_nao_atualizei(self):
        self.assertEqual(
            classify_response_intent("Não atualizei a configuração."),
            "answer",
        )

    def test_phantom_negation_en(self):
        self.assertEqual(
            classify_response_intent("I didn't create any file."),
            "answer",
        )

    def test_phantom_negation_en_not_updated(self):
        self.assertEqual(
            classify_response_intent("The configuration was not updated."),
            "answer",
        )

    def test_phantom_negation_pt_nunca(self):
        self.assertEqual(
            classify_response_intent("Nunca modifiquei esse código."),
            "answer",
        )

    # ------------------------------------------------------------------
    # Phantom action – questions (should NOT be phantom)
    # ------------------------------------------------------------------
    def test_phantom_question_pt(self):
        self.assertEqual(
            classify_response_intent("Você quer que eu crie o arquivo?"),
            "answer",
        )

    def test_phantom_question_en(self):
        self.assertEqual(
            classify_response_intent("Should I create the file?"),
            "answer",
        )

    # ------------------------------------------------------------------
    # Phantom action – with code (should NOT be phantom)
    # ------------------------------------------------------------------
    def test_phantom_with_code_block(self):
        self.assertEqual(
            classify_response_intent(
                "Criei o arquivo:\n```python\ndef foo():\n    pass\n```"
            ),
            "answer",
        )

    # ------------------------------------------------------------------
    # Answer patterns (should NOT be re-prompted)
    # ------------------------------------------------------------------
    def test_answer_code_block(self):
        response = (
            "```python\ndef foo():\n    pass\n```\nThe function is defined above."
        )
        self.assertEqual(classify_response_intent(response), "answer")

    def test_answer_with_summary_marker(self):
        self.assertEqual(
            classify_response_intent(
                "Resumo: O arquivo foi encontrado e contém 10 linhas."
            ),
            "answer",
        )

    def test_answer_with_summary_en(self):
        self.assertEqual(
            classify_response_intent(
                "Summary: The file was found and contains 10 lines."
            ),
            "answer",
        )

    def test_answer_with_file_path(self):
        self.assertEqual(
            classify_response_intent(
                "O conteúdo do file_path src/main.py mostra que a função está correta."
            ),
            "answer",
        )

    def test_answer_long_response(self):
        # Long responses are almost certainly real answers.
        long = (
            "After analyzing the codebase, I found that the issue is in "
            "src/services/process.py. The function `connectToServer` at line "
            "712 has a bug where it doesn't handle the timeout case properly. "
            "When the connection times out, it should retry with exponential "
            "backoff, but instead it raises the error directly. The fix is to "
            "wrap the socket.connect call in a try-except block and implement "
            "the retry logic. Here's the corrected code:"
            "\n```python\nimport time\n\ndef connect():\n    pass\n```"
        )
        self.assertEqual(classify_response_intent(long), "answer")

    def test_answer_direct_answer(self):
        self.assertEqual(
            classify_response_intent(
                "O arquivo src/main.py contém a classe Agent que implementa "
                "o loop principal do agente. Ele usa o provider para inferir "
                "respostas e dispatcha chamadas de ferramenta."
            ),
            "answer",
        )


class TestShouldReprompt(unittest.TestCase):
    def test_empty_response(self):
        reprompt, reason = should_reprompt("")
        self.assertTrue(reprompt)
        self.assertIn("empty", reason.lower())

    def test_thinking_response(self):
        reprompt, reason = should_reprompt(
            "Vou verificar o conteúdo do arquivo principal."
        )
        self.assertTrue(reprompt)
        self.assertIn("described what you would do", reason.lower())

    def test_planning_response(self):
        reprompt, reason = should_reprompt(
            "Meu plano é analisar os arquivos e depois fazer as alterações."
        )
        self.assertTrue(reprompt)
        self.assertIn("plan", reason.lower())

    def test_tool_hint_response(self):
        reprompt, reason = should_reprompt("Vou usar read_file para ver o conteúdo.")
        self.assertTrue(reprompt)
        self.assertIn("tool", reason.lower())

    def test_stalling_response(self):
        reprompt, reason = should_reprompt("Hmm...")
        self.assertTrue(reprompt)
        self.assertIn("filler", reason.lower())

    def test_phantom_action_response(self):
        reprompt, reason = should_reprompt(
            "Criei o arquivo main.py com o conteúdo correto."
        )
        self.assertTrue(reprompt)
        self.assertIn("claimed", reason.lower())

    def test_phantom_action_en_response(self):
        reprompt, reason = should_reprompt(
            "I've created the configuration file successfully."
        )
        self.assertTrue(reprompt)
        self.assertIn("claimed", reason.lower())

    def test_answer_not_reprompted(self):
        reprompt, reason = should_reprompt(
            "O arquivo src/main.py implementa o loop principal do agente "
            "que usa o provider para inferir respostas e dispatcha chamadas."
        )
        self.assertFalse(reprompt)
        self.assertIsNone(reason)

    def test_answer_with_code_not_reprompted(self):
        reprompt, reason = should_reprompt(
            "Aqui está o código corrigido:\n```python\ndef foo():\n    pass\n```"
        )
        self.assertFalse(reprompt)
        self.assertIsNone(reason)

    def test_is_tool_attempt_integration(self):
        # A malformed tool call should be detected as tool_hint.
        reprompt, reason = should_reprompt(
            '<|tool_call|>read_file file_path="src/main.py"'
        )
        self.assertTrue(reprompt)
        self.assertIn("tool", reason.lower())


if __name__ == "__main__":
    unittest.main()
