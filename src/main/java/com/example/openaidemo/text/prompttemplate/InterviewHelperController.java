package com.example.openaidemo.text.prompttemplate;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import com.example.openaidemo.services.OpenAiService;

@Controller
public class InterviewHelperController {
	@Autowired
	private OpenAiService openAiService;

	@GetMapping("/showInterviewHelper")
	public String showInterviewHelper() {
		return "interviewHelper";
	}

	@PostMapping("/interviewHelper")
	public String interviewHelper(@RequestParam("company") String company,
			@RequestParam("jobTitle") String jobTitle,
			@RequestParam("strength") String strength, 
			@RequestParam("weakness") String weakness, Model model) {

		String response = openAiService.getInteviewHelp(company, jobTitle, strength, weakness);

		model.addAttribute("response", response);
		return "interviewHelper";
	}

}
