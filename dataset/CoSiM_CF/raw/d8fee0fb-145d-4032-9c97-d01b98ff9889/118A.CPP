// Problem Link https://codeforces.com/problemset/problem/118/A
#include <iostream>
#include <string>

using namespace std;

int main() {
  string str;
  cin >> str;

  char c;
  for (int i = 0; i < str.length(); i++) {
    c = tolower(str[i]);
    if (string("aeiouy").find(c) != string::npos) {
      continue;
    } else {
      cout << "." << c;
    }
  }

  cout << "\n";
}