#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

int main() {
  string line, output = "";
  cin >> line;
  vector<int> numbers;
  for (auto chr : line) {
    if (chr != '+') {
      numbers.push_back(chr);
    }
  }
  std::sort(numbers.begin(), numbers.end());

  for (decltype(numbers.size()) i = 0; i != numbers.size(); ++i) {
    cout << static_cast<char>(numbers[i]);
    if (i != numbers.size() - 1) {
      cout << '+';
    }
  }

  cout << output;
  return 0;
}
