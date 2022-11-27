#include <cstdio>

int main() {
  int input;
  scanf("%d", &input);

  bool ans = false;
  for (int i = 2; i < input; i += 2) {
    if ((input - i) % 2 == 0) {
      ans = true;
      break;
    }
  }

  if (ans == false)
    printf("NO\n");
  else
    printf("YES\n");

  return 0;
}
