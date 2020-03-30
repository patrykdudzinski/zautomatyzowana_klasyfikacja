$('document').ready(function(){
    if($('div#parameters_list li').length > 0){
        $('#parameters_list').show();
    }
})


$('#send_file').on('click', function(){
    setTimeout(function(){
        var file = $('#fileuploader').val().split('\\');
        $.ajax({
            url: "/_get_file_data/",
            data: {'file' : file[file.length-1]},
            type: "POST",
            success: function(resp){
                $('#parameters_list').show();
                $('div#parameters_list').html(resp.data);
            }
        });
    }, 500)
})    


$('#parameters_list li').on('click', function(){
    var attr = $(this).text();
    var filename = $('#parameters_filename').text()+'.arff';
    var confirmation = confirm('Prognozować zmienną '+attr+'?')
    if(confirmation === true){
        $.ajax({
            url: "/_forecast_data/",
            data: {'filename' : filename,
                   'attr': attr},
            type: "POST",
            success: function(resp){
                 $('div#chart').html(resp.data);
                //$('#parameters_list').show();
                //$('div#parameters_list').html(resp.data);
            }
        });            
    }
})