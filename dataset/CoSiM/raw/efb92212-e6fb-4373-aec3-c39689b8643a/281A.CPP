#include <bits/stdc++.h>

typedef long long ll;

using namespace std;

int main() {
  string input_string;

  cin >> input_string;

  if (input_string[0] > 90)
    input_string[0] -= 32;

  cout << input_string << endl;
}

// g++ -std=c++11 -O2 -Wall 281A.cpp -o out