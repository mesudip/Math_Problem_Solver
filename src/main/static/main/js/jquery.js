$(function(){

    var catagory="";
    var searchBy="";
    function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie != '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) == (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
    var csrftoken = getCookie('csrftoken');
    $('.sortby').click(function(){
        if($(this).text() == "Reset"){
            $('.sorttoggle').html("Sort By"+ "<span class='caret'></span>");
        }
        else{
            $('.sorttoggle').html("Sorted By: " + $(this).text()+ "<span class='caret'></span>");
        }

        searchBy = $(this).text();

        if(searchBy == "Price"){
            searchBy="P";
        }
        else if(searchBy == "A-Z"){
            searchBy="A";
        }
        else if(searchBy == "Manufacture Date"){
            searchBy="D";
        }
        else{
            searchBy=""
        }
        console.log(searchBy);
        $('.content').html('').load("/ajaxSearch/?search=" + encodeURIComponent($('#searchbar').val()) + "&catagory=" + catagory + "&by=" + searchBy);

    });

    $('.navCat').click(function(){
        $('.navCat').parent().removeClass("active");
        $(this).parent().addClass("active");

        catagory = $(this).text();

        if(catagory == "Agricultural Products"){
            catagory="L";
        }
        else if(catagory == "Machineries"){
            catagory="M";
        }
        console.log(catagory);
        console.log(searchBy);
        $('.content').html('').load("/ajaxSearch/?search=" + encodeURIComponent($('#searchbar').val()) + "&catagory=" + catagory + "&by=" + searchBy);
    });

    $("#searchForm").on('submit',function(event){
        event.preventDefault();
    });

    $("#searchbar").keyup(function(){
        $('.content').html('').load("/ajaxSearch/?search=" + encodeURIComponent($(this).val()) + "&catagory=" + catagory + "&by=" + searchBy);
    });

    $('.showdiv').click(function(){
        $('.warranty').toggleClass('hidden');
    });
    $("#searchform").on('submit',function(event){
        event.preventDefault();
        var answer;
        $.ajax({
       type: "POST",
       url: "/api/entry/",
       data: {
           'csrfmiddlewaretoken':csrftoken,
           'question': $('#search').val(), // < note use of 'this' here
       },
       success: function(result) {
           answer = result.answer;
           $('#answer').text(answer);
       },
       error: function(result) {
           alert('recent question update failed')
       }
   });
        $("#answerdiv").removeClass('hidden');
    });
    $('#incorrect').click(function(event){
        $('#selectanswer').removeClass('hidden');
    });
    $("#search").bind('input', function(){
        $('#answer').text('');
        $('#selectanswer').addClass('hidden');
        $("#answerdiv").addClass('hidden');

    });

});
