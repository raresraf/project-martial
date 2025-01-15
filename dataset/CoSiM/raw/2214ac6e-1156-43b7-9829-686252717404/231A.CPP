// https://codeforces.com/problemset/problem/231/A
#include <iostream>
using namespace std;
int main() {
  int num_prob, petya, vasya, tonya, result = 0;
  cin >> num_prob;
  for (int i = 0; i < num_prob; i++) {
    cin >> petya >> vasya >> tonya;
    if (petya + vasya + tonya >= 2) {
      result += 1;
    }
  }
  cout << result << endl;
}
