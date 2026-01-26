#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string expression, result;
  cin >> expression;
  int length = expression.length();

  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '1')
      result.append("1+");
  }

  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '2')
      result.append("2+");
  }

  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '3')
      result.append("3+");
  }

  result.pop_back();
  cout << result;

  return 0;
}