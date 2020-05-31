$('#form_nb_probas').on('change', function(){
    $("#error_message").text("");
    $(this).removeClass("is-invalid-input");
    var probes_qty = $(this).val().split(',');
    var class_qty = parseInt($('#class_names_length').val());
    if(probes_qty.length !== class_qty){
        $("#error_message").text("Prawdopodobieństwa nie równe liczbie klas = "+class_qty);
        $(this).addClass("is-invalid-input");
    }
})