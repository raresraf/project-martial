/*
 * Link: https://codeforces.com/problemset/problem/231/A
 */

#include <iostream>
using namespace std;

int main() {
  int number_of_problems;
  cin >> number_of_problems;
  int probs_to_submit = 0;
  for (int i = 0; i < number_of_problems; i++) {
    int temp, line_sum = 0;
    for (int j = 0; j < 3; j++) {
      cin >> temp;
      line_sum += temp;
    }
    if (line_sum > 1)
      probs_to_submit++;
  }
  cout << probs_to_submit << endl;
}