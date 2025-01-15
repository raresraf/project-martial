#include <iostream>
#include <string>

using namespace std;

struct Expression {
  int one, two, three, total;
  Expression() : one{0}, two{0}, three{0}, total{0} {};
};

int main() {
  string expression;
  cin >> expression;
  Expression expr = Expression();
  for (size_t i = 0; i < expression.size(); i += 2) {
    if (expression[i] == '1')
      ++expr.one;
    else if (expression[i] == '2')
      ++expr.two;
    else
      ++expr.three;
    ++expr.total;
  }

  for (int i = 0; i < expr.total; ++i) {
    if (i != 0)
      cout << '+';
    if (expr.one > 0) {
      cout << 1;
      --expr.one;
    } else if (expr.two > 0) {
      cout << 2;
      --expr.two;
    } else {
      cout << 3;
      --expr.three;
    }
  }
  cout << endl;

  return 0;
}
