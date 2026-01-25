#include <bits/stdc++.h>
#include <iostream>

int main() {
  using namespace std;
  string addition;
  cin >> addition;
  vector<int> numbers;
  for (auto i = 0; i < (int)addition.length(); i += 2)
    numbers.push_back(addition.at(i) - '0');
  sort(numbers.begin(), numbers.end());
  string final_str = "";
  for (auto i = 0; i < (int)addition.length(); i++) {
    if (i % 2 == 0)
      final_str += (numbers.at(i / 2) + '0');
    else
      final_str += '+';
  }
  cout << final_str << endl;
  return 0;
}
