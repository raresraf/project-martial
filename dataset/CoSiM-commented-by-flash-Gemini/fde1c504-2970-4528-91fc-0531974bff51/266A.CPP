#include <cstring>
#include <iostream>

int main(int argc, char const *argv[]) {
  int i, n, num = 0;
  char str[100], temp;
  std::cin >> n;
  std::cin >> str;
  temp = str[0];
  for (i = 1; i < n; i++) {
    if (str[i] == temp)
      num++;
    else
      temp = str[i];
  }
  std::cout << num << '\n';
  return 0;
}
