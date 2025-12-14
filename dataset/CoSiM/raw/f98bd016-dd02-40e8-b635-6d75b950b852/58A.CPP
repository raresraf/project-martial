#include <cstring>
#include <iostream>

int main() {
  int i, j, flag = 0;
  char str[103];
  char str2[5] = {'h', 'e', 'l', 'l', 'o'};
  std::cin >> str;
  for (i = 0, j = 0; i < strlen(str); i++) {
    if (str[i] == str2[j])
      j++;
    if (j == 5) {
      flag = 1;
      break;
    }
  }
  if (flag == 1)
    std::cout << "YES" << '\n';
  else if (flag == 0)
    std::cout << "NO" << '\n';
  return 0;
}
