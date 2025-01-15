#include <bits/stdc++.h>
using namespace std;

int main() {
  string input;
  cin >> input;
  int d = 1;
  sort(input.begin(), input.end());
  char t = input[0];
  for (int i = 1; i < input.size(); i++) {
    if (t != input[i]) {
      d++;
      t = input[i];
    }
  }
  if (d % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
}
