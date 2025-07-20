// http://codeforces.com/problemset/problem/339/A

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

int main() {
  string s, token;
  cin >> s;
  stringstream ss(s);
  vector<string> Xenia;
  while (getline(ss, token, '+')) {
    Xenia.push_back(token);
  }
  sort(Xenia.begin(), Xenia.end());
  for (int i = 0; i < Xenia.size() - 1; i++)
    cout << Xenia[i] << "+";
  cout << Xenia[Xenia.size() - 1];
  return 0;
}
