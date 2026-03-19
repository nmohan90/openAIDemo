package com.example.openaidemo.services;

import com.example.openaidemo.text.prompttemplate.dto.CountryCuisines;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class OpenAiService {

    private ChatClient chatClient;

    @Autowired
    public EmbeddingModel embeddingModel;

    public OpenAiService (ChatClient.Builder builder, ChatMemory chatMemory) {
        chatClient = builder.defaultAdvisors(MessageChatMemoryAdvisor.builder(chatMemory).build()).build();
    }


    public ChatResponse askAnything(String question) {
        ChatResponse response = chatClient.prompt().user(question).call().chatResponse();
        return response;
    }

    public String getTravelResponse(String city, String month, String language, String budget) {

        PromptTemplate promptTemplate = new PromptTemplate("Welcome to the {city} travel guide!\n" +
                "If you're visiting in {month}, here's what you can do:\n" +
                "1. Must-visit attractions.\n" +
                "2. Local cuisine you must try.\n" +
                "3. Useful phrases in {language}.\n" +
                "4. Tips for traveling on a {budget} budget.\n" +
                "Enjoy your trip!");

        Prompt prompt =promptTemplate.create(Map.of("city", city,"month", month, "language", language, "budget", budget));

        String response = chatClient.prompt(prompt).call().chatResponse().getResult().getOutput().getText();

        return response;
    }

    public CountryCuisines getCuisines(String country, String numCuisines, String language) {

        PromptTemplate promptTemplate = new PromptTemplate("You are an expert in traditional cuisines.\n" +
                "You provide information about a specific dish from a specific\n" +
                "country.\n" +
                "Answer the question: What is the traditional cuisine of {country}" +
                "Return a list of {numCuisines} in {language}" +
                "Avoid giving information about fictional places. If the country is\n" +
                "fictional\n" +
                "or non-existent answer: I don't know");

        Prompt prompt =promptTemplate.create(Map.of("country", country,"numCuisines", numCuisines, "language", language));

        System.out.println("country: " +country);
        System.out.println("language" + language);
        CountryCuisines response = chatClient.prompt(prompt).call().entity(CountryCuisines.class);

        return response;
    }

    public String getInteviewHelp(String company, String jobTitle, String strength, String weakness) {

        PromptTemplate promptTemplate = new PromptTemplate("You are applying for a new {company}" +
                "The job role that you are applying is {jobTitle}" +
                "Here are your {strength} and {weakness}" +
                        "Based on which this provide guidance how to should prepare for the interview!" +
                        "All the besst");

        Prompt prompt =promptTemplate.create(Map.of("company", company,"jobTitle", jobTitle, "strength", strength, "weakness", weakness));

        String response = chatClient.prompt(prompt).call().chatResponse().getResult().getOutput().getText();

        return response;
    }

    public float[] embedding(String text) {
        return embeddingModel.embed(text);
    }

    public double findSimilarity(String text1, String text2) {

        List<String> list = new ArrayList<>();
        list.add(text1);
        list.add(text2);

        List<float[]> response = embeddingModel.embed(List.of(text1,text2));
        return cosineSimilarity(response.get(0), response.get(1));
    }



    private double cosineSimilarity(float[] vectorA, float[] vectorB) {
        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException("Vectors must be of the same length");
        }

        // Initialize variables for dot product and magnitudes
        double dotProduct = 0.0;
        double magnitudeA = 0.0;
        double magnitudeB = 0.0;

        // Calculate dot product and magnitudes
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            magnitudeA += vectorA[i] * vectorA[i];
            magnitudeB += vectorB[i] * vectorB[i];
        }

        // Calculate and return cosine similarity
        return dotProduct / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
    }
}
