package com.example.openaidemo.text;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RestController;

import com.example.openaidemo.services.OpenAiService;

@RestController
public class AnswerAnyThingStreamingController {

	@Autowired
	OpenAiService service;

}