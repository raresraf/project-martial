#include <cstring>
#include <iostream>
using namespace std;

int main() {
  int loop_count;
  cin >> loop_count;

  string input;
  int result = 0;
  while (loop_count--) {
    cin >> input;
    if (input == "++X" || input == "X++")
      result++;
    else
      result--;
  }
  cout << result;
  return 0;
}