/*
    desc: Walidacja parametrów formularza
    @param chosen_method - metoda uczenia pobrana z selectboxa form_method
    @return boolean true/false
*/
function check_parmeters(chosen_method) {
    if (chosen_method === "") {
        $("#error_message").text("Wybierz metodę uczenia");
        $("#form_method").addClass("is-invalid-input");
        return false;
    }

    if ($("#form_iterations").val() === "") {
        $("#form_iterations").addClass("is-invalid-input");
        $("#error_message").text("Uzupełnij liczbę iteracji");
        return false;
    }

    if ($("#form_score").val() === "") {
        $("#form_score").addClass("is-invalid-input");
        $("#error_message").text("Uzupełnij minimalną dokładność");
        return false;
    }

    $(".additional_parameters.visible .form_option:not(.hidden)").each(function(
        key,
        item
    ) {
        if ($(item).val() === "") {
            $(item).addClass("is-invalid-input");
            $("#error_message").text("Uzupełnij wartości parametrów");
            return false;
        }
    });

    if ($("#form_lr_l1_ratio").hasClass("hidden")) {
        $("#form_lr_l1_ratio").val("0");
    }

    return true;
}
