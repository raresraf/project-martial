#include <bits/stdc++.h>

using namespace std;

int main() {
  int ans = 0;
  int number_input;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> number_input;

      if (number_input == 1) {
        ans += abs(j - 2) + abs(i - 2);
      }
    }
  }
  cout << ans << "\n";
}

// g++ -std=c++11 -O2 -Wall 263A.cpp -o out