#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  vector<int> numbers;
  string form;
  cin >> form;
  for (int i = 0; i < form.length() / 2 + 1; i++) {
    int x = form.at(i * 2);
    numbers.push_back(x - 48);
  }
  sort(numbers.begin(), numbers.end());
  for (vector<int>::iterator it = numbers.begin(); it != numbers.end(); ++it) {
    if (it == numbers.begin())
      printf("%d", *it);
    else
      printf("+%d", *it);
  }
  return 0;
}
