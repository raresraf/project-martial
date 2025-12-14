#include <bits/stdc++.h>
using namespace std;

int main() {
  string input;
  int count;
  cin >> input;

  sort(input.begin(), input.end());

  auto result = unique(input.begin(), input.end());
  input = string(input.begin(), result);

  count = input.length();
  if (count % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;

  return 0;
}