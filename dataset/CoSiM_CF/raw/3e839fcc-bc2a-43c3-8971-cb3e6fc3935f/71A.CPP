// https://codeforces.com/problemset/problem/71/A
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n;
  string input;
  string output;

  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> input;
    if (input.size() > 10) {
      input = input[0] + to_string(input.size() - 2) + input.back();
    }
    if (i < n - 1) {
      output += input + "\n";
    } else {
      output += input;
    }
  }
  cout << output;
}