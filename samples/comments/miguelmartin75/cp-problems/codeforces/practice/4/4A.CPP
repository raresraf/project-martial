#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
  int n;
  std::cin >> n;

  std::cout << (n > 2 && n % 2 == 0 ? "YES" : "NO") << std::endl;
}
