// http://codeforces.com/problemset/problem/236/A
#include <bits/stdc++.h>
using namespace std;

int getDistinct(string input) {
  // returns the number of characters in a string
  map<char, int> counts;
  int count = 0;
  for (int i = 0; i < input.size(); i++) {
    if (counts.find(input[i]) == counts.end()) {
      // if this character is not already in the map, add it and set count to 1
      counts.insert(pair<char, int>(input[i], 1));
      count++;
    }
  }
  return count;
}

int main() {
  string username;
  cin >> username;
  int count = getDistinct(username);
  if (count % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  cout << endl;
  return 0;
}
