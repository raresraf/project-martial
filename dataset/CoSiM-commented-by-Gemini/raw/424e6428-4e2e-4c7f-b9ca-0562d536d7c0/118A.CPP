#include <bits/stdc++.h>
using namespace std;

int main() {
  string task;
  cin >> task;
  // task = tolower(task);
  string result = "";
  set<char> vowels = {'a', 'e', 'i', 'o', 'u', 'y'};
  for (auto s : task) {
    s = tolower(s);
    if (s >= 'b' && s <= 'z' && vowels.find(s) == vowels.end()) {
      result += '.';
      result += s;
    }
  }
  cout << result << "\n";
}