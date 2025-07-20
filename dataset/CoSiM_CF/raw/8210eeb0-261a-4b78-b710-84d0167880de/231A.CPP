#include <iostream>
using namespace std;

int main() {
  int line_count;
  cin >> line_count;

  int member1, member2, member3, count = 0;

  while (line_count--) {
    cin >> member1 >> member2 >> member3;
    if (member1 + member2 + member3 >= 2)
      count++;
  }
  cout << count;
  return 0;
}