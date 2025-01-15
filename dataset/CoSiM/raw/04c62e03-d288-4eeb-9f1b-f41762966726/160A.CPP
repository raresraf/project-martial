#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, sum = 0, suma = 0, c = 0;
  std::cin >> n;
  int k;
  std::vector<int> x;
  for (int i = 0; i < n; i++) {
    std::cin >> k;
    x.push_back(k);
    sum += k;
  }
  std::sort(x.rbegin(), x.rend());
  for (int j = 0; suma <= sum / 2; j++) {
    suma += x[j];
    c++;
  }
  std::cout << c;
  return 0;
}