#!/usr/bin/env python3
"""
Test script for Rice Disease RAG Service
This script tests the chat endpoint with sample queries.
"""

import requests
import json
import sys

API_URL = "http://localhost:4000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("🏥 Testing Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chat(query, description):
    """Test the chat endpoint with a query"""
    print("\n" + "="*60)
    print(f"💬 Testing: {description}")
    print("="*60)
    print(f"Query: {query}")
    
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Response:")
            print(f"Language: {result['language']}")
            print(f"Answer: {result['answer']}")
            
            if result.get('translated_query'):
                print(f"Translated Query: {result['translated_query']}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_stats():
    """Test the stats endpoint"""
    print("\n" + "="*60)
    print("📊 Testing Stats Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/stats")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🌾 Rice Disease RAG Service - Test Suite")
    print("="*60)
    print(f"API URL: {API_URL}")
    
    # Check if service is running
    try:
        response = requests.get(API_URL)
        if response.status_code != 200:
            print("\n❌ Service is not running. Please start the service first:")
            print("   uvicorn main:app --port 4000")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to the service. Please start the service first:")
        print("   uvicorn main:app --port 4000")
        sys.exit(1)
    
    print("\n✅ Service is running!")
    
    # Run tests
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health_check()))
    
    # Test 2: English query about symptoms
    results.append((
        "English Query - Symptoms",
        test_chat(
            "What are the symptoms of brown spot disease?",
            "English Query - Brown Spot Symptoms"
        )
    ))
    
    # Test 3: English query about treatment
    results.append((
        "English Query - Treatment",
        test_chat(
            "How can I treat bacterial leaf blight?",
            "English Query - Treatment"
        )
    ))
    
    # Test 4: English query about prevention
    results.append((
        "English Query - Prevention",
        test_chat(
            "How do I prevent rice blast?",
            "English Query - Prevention"
        )
    ))
    
    # Test 5: Bangla query (if translation is working)
    results.append((
        "Bangla Query",
        test_chat(
            "ব্রাউন স্পট রোগের লক্ষণ কি?",
            "Bangla Query - Brown Spot Symptoms"
        )
    ))
    
    # Test 6: Stats endpoint
    results.append(("Stats Endpoint", test_stats()))
    
    # Print summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
