// CodeForces
// Problem 339A [Helpful Maths]

#include <bits/stdc++.h>
using namespace std;

int main() {
  string str1;
  cin >> str1;

  vector<int> numbers;
  for (unsigned int copystrindex = 0; copystrindex < str1.length();
       copystrindex++) {
    if (str1[copystrindex] == '+') {
      continue;
    } else {
      numbers.push_back(str1[copystrindex] - '0');
    }
  }
  sort(numbers.begin(), numbers.end());

  for (unsigned int outputindex = 0; outputindex < numbers.size();
       outputindex++) {
    if (outputindex != (numbers.size() - 1)) {
      cout << numbers[outputindex] << "+";
    } else {
      cout << numbers[outputindex];
    }
  }
  return 0;
}