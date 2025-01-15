#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  int n;
  int temp;
  int minNumber = 0;

  int totalSum = 0;
  int mySum = 0;

  std::vector<int> cont;

  std::cin >> n;

  // push to vector
  for (int i = 0; i < n; i++) {
    std::cin >> temp;
    totalSum += temp;

    cont.push_back(temp);
  }

  // sort
  std::sort(cont.begin(), cont.end());

  // reverse
  std::reverse(cont.begin(), cont.end());

  // find minimal coin amount
  for (int i = 0; i < n; i++) {
    minNumber++;

    mySum += cont[i];

    if (mySum > totalSum - mySum) {
      break;
    }
  }

  std::cout << minNumber << std::endl;

  return 0;
}
