#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  string s;
  cin >> s;
  vector<char> numbers;
  for (int i = 0; i < s.length(); ++i) {
    if (s[i] != '+')
      numbers.push_back(s[i]);
  }
  sort(numbers.begin(), numbers.end());

  int numbersLen = numbers.size();
  for (int i = 0; i < numbersLen - 1; ++i)
    cout << numbers[i] << '+';

  cout << numbers[numbersLen - 1] << "\n";
  return 0;
}