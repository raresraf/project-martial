#include <bits/stdc++.h>

using namespace std;

int main() {
  string first, second;

  cin >> first >> second;

  transform(first.begin(), first.end(), first.begin(), ::tolower);
  transform(second.begin(), second.end(), second.begin(), ::tolower);

  int size_of_first = first.size();
  int size_of_second = second.size();
  int ans = 0;

  for (int i = 0; i < size_of_first; i++) {
    if (first[i] > second[i]) {
      ans = 1;
      break;
    } else if (second[i] > first[i]) {
      ans = -1;
      break;
    }
  }

  cout << ans << "\n";
}

// g++ -std=c++11 -O2 -Wall 112A.cpp -o out