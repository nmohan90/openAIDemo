package com.example.openaidemo.text.prompttemplate;

import com.example.openaidemo.text.prompttemplate.dto.CountryCuisines;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import com.example.openaidemo.services.OpenAiService;

@Controller
public class CuisineHelperController {
	@Autowired
    private OpenAiService openAiService;

    @GetMapping("/showCuisineHelper")
    public String showChatPage() {
         return "cuisineHelper";
    }

    @PostMapping("/cuisineHelper")
    public String getChatResponse(@RequestParam("country") String country, @RequestParam("numCuisines") String numCuisines,@RequestParam("language") String language,Model model) {

        CountryCuisines countryCuisines = openAiService.getCuisines(country, numCuisines, language);

        model.addAttribute("countryCuisines",countryCuisines);
        return "cuisineHelper";
    }
}
