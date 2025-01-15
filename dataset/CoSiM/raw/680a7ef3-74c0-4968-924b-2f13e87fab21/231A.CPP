/**
 * Subject: Team
 * URL: https://codeforces.com/problemset/problem/231/A
 */
#include <iostream>

int main() {
  std::ios::sync_with_stdio(0);
  std::cin.tie(0);
  std::cout.tie(0);

  int noOfProblems;
  std::cin >> noOfProblems;
  int counter = 0;
  while (noOfProblems--) {
    int a, b, c;
    std::cin >> a >> b >> c;
    int sum = a + b + c;
    if (sum >= 2)
      counter++;
  }
  std::cout << counter << "\n";
  return 0;
}