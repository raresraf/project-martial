#include <stdio.h>
#include <string.h>

int main(void) {

  char s[100] = {
      0,
  };
  char n[100] = {
      0,
  };
  int n_idx = 1;

  scanf("%s", s);

  for (int idx_h = 0; s[idx_h] != '\0'; idx_h++) {
    if (s[idx_h] == 'h') {
      for (int idx_e = idx_h; s[idx_e] != '\0'; idx_e++) {
        if (s[idx_e] == 'e') {
          for (int idx_l = idx_e; s[idx_l] != '\0'; idx_l++) {
            if (s[idx_l] == 'l') {
              for (int idx_l2 = idx_l + 1; s[idx_l2] != '\0'; idx_l2++) {
                if (s[idx_l2] == 'l') {
                  for (int idx_o = idx_l2; s[idx_o] != '\0'; idx_o++) {
                    if (s[idx_o] == 'o') {
                      printf("YES");
                      return 0;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  printf("NO");
  return 0;
}
