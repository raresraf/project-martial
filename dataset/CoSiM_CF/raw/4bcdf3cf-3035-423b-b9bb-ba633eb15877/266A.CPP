#include <iostream>
#include <stack>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  std::stack<char> table;
  string input;
  int n, removeCounter = 0;
  cin >> n;

  cin >> input;

  for (auto stone : input) {
    if (!table.empty() && table.top() == stone) {
      removeCounter++;
    } else {
      table.push(stone);
    }
  }

  cout << removeCounter;
  return 0;
}
